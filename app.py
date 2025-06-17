import gradio as gr
import subprocess
import tempfile
import os
import shutil
from pathlib import Path
import torch
import torchaudio
import numpy as np
from df import enhance, init_df
import logging
import warnings

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="df")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAudioEnhancer:
    def __init__(self):
        """DeepFilterNet 모델 초기화"""
        try:
            logger.info("DeepFilterNet 모델 로딩 중...")
            self.model, self.df_state, _ = init_df()
            logger.info("DeepFilterNet 모델 로딩 완료")
        except Exception as e:
            logger.error(f"DeepFilterNet 모델 로딩 실패: {e}")
            self.model = None
            self.df_state = None
    
    def extract_audio_from_video(self, video_path, audio_path):
        """1단계: 영상에서 오디오 추출 (FFmpeg 사용)"""
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # 비디오 스트림 제외
                '-acodec', 'pcm_s16le',  # WAV 포맷
                '-ar', '48000',  # 48kHz 샘플링 레이트 (DeepFilterNet 요구사항)
                '-ac', '1',  # 모노로 변환
                '-y',  # 덮어쓰기
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"오디오 추출 완료: {audio_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"오디오 추출 실패: {e.stderr}")
            return False
    
    def convert_mono_to_stereo(self, mono_audio_path, stereo_audio_path):
        """모노 오디오를 스테레오로 변환 (양쪽 채널에 동일한 소리)"""
        try:
            cmd = [
                'ffmpeg', '-i', mono_audio_path,
                '-ac', '2',  # 스테레오로 변환
                '-af', 'pan=stereo|c0=c0|c1=c0',  # 모노 채널을 양쪽으로 복사
                '-y',  # 덮어쓰기
                stereo_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"스테레오 변환 완료: {stereo_audio_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"스테레오 변환 실패: {e.stderr}")
            return False
    
    def enhance_audio_with_deepfilter(self, input_audio_path, output_audio_path):
        """2단계: DeepFilterNet으로 오디오 정제"""
        try:
            if self.model is None or self.df_state is None:
                raise Exception("DeepFilterNet 모델이 로드되지 않았습니다")
            
            try:
                # 오디오 로드 시도
                audio, sr = torchaudio.load(input_audio_path)
            except Exception as e:
                logger.warning(f"torchaudio 로드 실패, librosa로 시도: {e}")
                # librosa를 백업으로 사용
                import librosa
                audio_np, sr = librosa.load(input_audio_path, sr=None, mono=False)
                audio = torch.from_numpy(audio_np)
                if len(audio.shape) == 1:
                    audio = audio.unsqueeze(0)
            
            # 48kHz가 아닌 경우 리샘플링
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                audio = resampler(audio)
                sr = 48000
            
            # 모노로 변환 (필요한 경우)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # DeepFilterNet은 [channels, samples] 형태를 기대함
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            
            logger.info(f"처리 전 오디오 형태: {audio.shape}, 샘플레이트: {sr}")
            
            # 긴 오디오를 청크로 나누어 처리 (메모리 절약)
            chunk_size = 48000 * 30  # 30초 청크
            total_samples = audio.shape[1]
            
            if total_samples > chunk_size:
                logger.info(f"긴 오디오 감지 ({total_samples/48000:.1f}초). 청크 단위로 처리합니다.")
                enhanced_chunks = []
                
                for i in range(0, total_samples, chunk_size):
                    end_idx = min(i + chunk_size, total_samples)
                    chunk = audio[:, i:end_idx]
                    
                    # 메모리 정리
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    logger.info(f"청크 {i//chunk_size + 1} 처리 중... ({i/48000:.1f}s - {end_idx/48000:.1f}s)")
                    
                    try:
                        # 각 청크를 contiguous하게 만들기
                        chunk = chunk.contiguous()
                        
                        with torch.no_grad():
                            enhanced_chunk = enhance(self.model, self.df_state, chunk)
                        
                        # CPU로 이동하여 메모리 절약
                        if isinstance(enhanced_chunk, torch.Tensor):
                            enhanced_chunk = enhanced_chunk.cpu()
                        
                        enhanced_chunks.append(enhanced_chunk)
                        
                    except Exception as e:
                        logger.warning(f"청크 {i//chunk_size + 1} GPU 처리 실패, CPU로 시도: {e}")
                        # CPU로 폴백
                        try:
                            # 모델을 CPU로 이동
                            cpu_model, cpu_df_state, _ = init_df()
                            if hasattr(cpu_model, 'cpu'):
                                cpu_model = cpu_model.cpu()
                            
                            chunk_cpu = chunk.cpu()
                            with torch.no_grad():
                                enhanced_chunk = enhance(cpu_model, cpu_df_state, chunk_cpu)
                            
                            enhanced_chunks.append(enhanced_chunk)
                            
                        except Exception as cpu_e:
                            logger.error(f"CPU 처리도 실패: {cpu_e}")
                            # 원본 청크 사용 (처리 실패시)
                            enhanced_chunks.append(chunk.squeeze())
                
                # 모든 청크 결합
                if isinstance(enhanced_chunks[0], torch.Tensor):
                    enhanced_audio = torch.cat(enhanced_chunks, dim=-1)
                else:
                    enhanced_audio = np.concatenate(enhanced_chunks, axis=-1)
                
                logger.info(f"청크 처리 완료. 총 {len(enhanced_chunks)}개 청크 병합")
                
            else:
                # 짧은 오디오는 한 번에 처리
                try:
                    # 메모리 연속성 확보
                    audio = audio.contiguous()
                    
                    with torch.no_grad():
                        enhanced_audio = enhance(self.model, self.df_state, audio)
                        
                except Exception as e:
                    logger.warning(f"GPU 처리 실패, CPU로 폴백: {e}")
                    # CPU 폴백
                    cpu_model, cpu_df_state, _ = init_df()
                    if hasattr(cpu_model, 'cpu'):
                        cpu_model = cpu_model.cpu()
                    
                    audio_cpu = audio.cpu()
                    with torch.no_grad():
                        enhanced_audio = enhance(cpu_model, cpu_df_state, audio_cpu)
            
            logger.info(f"처리 후 오디오 형태: {enhanced_audio.shape if hasattr(enhanced_audio, 'shape') else type(enhanced_audio)}")
            
            # 결과 저장 준비
            if isinstance(enhanced_audio, torch.Tensor):
                if len(enhanced_audio.shape) == 1:
                    enhanced_tensor = enhanced_audio.unsqueeze(0)
                else:
                    enhanced_tensor = enhanced_audio
            else:
                # numpy 배열인 경우
                if len(enhanced_audio.shape) == 1:
                    enhanced_tensor = torch.from_numpy(enhanced_audio).unsqueeze(0)
                else:
                    enhanced_tensor = torch.from_numpy(enhanced_audio)
            
            # 저장 시도
            try:
                torchaudio.save(output_audio_path, enhanced_tensor, sr)
                logger.info(f"torchaudio로 저장 성공: {output_audio_path}")
            except Exception as e:
                logger.warning(f"torchaudio 저장 실패, soundfile로 시도: {e}")
                import soundfile as sf
                if isinstance(enhanced_audio, torch.Tensor):
                    audio_to_save = enhanced_audio.squeeze().cpu().numpy()
                else:
                    audio_to_save = enhanced_audio.squeeze() if enhanced_audio.ndim > 1 else enhanced_audio
                sf.write(output_audio_path, audio_to_save, sr)
                logger.info(f"soundfile로 저장 성공: {output_audio_path}")
            
            logger.info(f"오디오 정제 완료: {output_audio_path}")
            return True
            
        except Exception as e:
            logger.error(f"오디오 정제 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def replace_audio_in_video(self, original_video_path, enhanced_audio_path, output_video_path):
        """3단계: 정제된 오디오를 원본 영상에 삽입"""
        try:
            cmd = [
                'ffmpeg', '-i', original_video_path,
                '-i', enhanced_audio_path,
                '-c:v', 'copy',  # 비디오 스트림 복사 (재인코딩 없음)
                '-c:a', 'aac',   # 오디오를 AAC로 인코딩
                '-map', '0:v:0',  # 첫 번째 입력의 비디오 스트림
                '-map', '1:a:0',  # 두 번째 입력의 오디오 스트림
                '-shortest',      # 가장 짧은 스트림에 맞춤
                '-y',             # 덮어쓰기
                output_video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"영상 생성 완료: {output_video_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"영상 생성 실패: {e.stderr}")
            return False

# 전역 enhancer 인스턴스
enhancer = VideoAudioEnhancer()

def process_video(video_file, use_postfilter=False, convert_to_stereo=False):
    """메인 처리 함수"""
    if video_file is None:
        return None, "영상 파일을 업로드해주세요."
    
    try:
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # 파일 경로 설정
            input_video_path = temp_dir / "input_video.mp4"
            extracted_audio_path = temp_dir / "extracted_audio.wav"
            enhanced_audio_path = temp_dir / "enhanced_audio.wav"
            stereo_audio_path = temp_dir / "stereo_audio.wav"
            output_video_path = temp_dir / "output_video.mp4"
            
            # 입력 파일 복사
            shutil.copy2(video_file, input_video_path)
            
            status_msg = "🎬 영상에서 오디오 추출 중..."
            
            # 1단계: 오디오 추출
            if not enhancer.extract_audio_from_video(str(input_video_path), str(extracted_audio_path)):
                return None, "❌ 오디오 추출에 실패했습니다."
            
            status_msg += "\n✅ 오디오 추출 완료\n🎧 DeepFilterNet으로 음성 정제 중..."
            
            # 2단계: 오디오 정제
            if not enhancer.enhance_audio_with_deepfilter(str(extracted_audio_path), str(enhanced_audio_path)):
                return None, "❌ 오디오 정제에 실패했습니다."
            
            status_msg += "\n✅ 음성 정제 완료"
            
            # 선택사항: 스테레오 변환
            final_audio_path = enhanced_audio_path
            if convert_to_stereo:
                status_msg += "\n🔊 모노 오디오를 스테레오로 변환 중..."
                if not enhancer.convert_mono_to_stereo(str(enhanced_audio_path), str(stereo_audio_path)):
                    return None, "❌ 스테레오 변환에 실패했습니다."
                final_audio_path = stereo_audio_path
                status_msg += "\n✅ 스테레오 변환 완료"
            
            status_msg += "\n📼 정제된 오디오를 영상에 삽입 중..."
            
            # 3단계: 영상에 오디오 삽입
            if not enhancer.replace_audio_in_video(str(input_video_path), str(final_audio_path), str(output_video_path)):
                return None, "❌ 영상 생성에 실패했습니다."
            
            status_msg += "\n✅ 영상 생성 완료!"
            
            # 결과 파일을 임시 위치에서 복사
            final_output_path = f"enhanced_{Path(video_file).stem}.mp4"
            shutil.copy2(output_video_path, final_output_path)
            
            return final_output_path, status_msg
            
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        return None, f"❌ 처리 중 오류가 발생했습니다: {str(e)}"

# Gradio 인터페이스 생성
def create_interface():
    with gr.Blocks(title="🎬 Video Audio Enhancement with DeepFilterNet", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 영상 음성 향상 도구
        
        DeepFilterNet3를 사용하여 영상의 배경 소음을 제거하고 음성을 향상시킵니다.
        
        ## 📋 처리 과정
        1. **🎥 오디오 추출**: 영상에서 오디오를 WAV 형식으로 추출
        2. **🎧 음성 정제**: DeepFilterNet3로 배경 소음 제거 및 음성 향상
        3. **📼 영상 생성**: 정제된 오디오를 원본 영상에 삽입하여 새로운 영상 생성
        
        ## 📝 지원 형식
        - **입력**: MP4, AVI, MOV, MKV 등 대부분의 영상 형식
        - **출력**: MP4 (H.264 비디오 + AAC 오디오)
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.File(
                    label="📁 영상 파일 업로드",
                    file_types=["video"],
                    file_count="single"
                )
                
                postfilter_checkbox = gr.Checkbox(
                    label="🔧 포스트필터 사용 (매우 시끄러운 섹션 추가 감쇠)",
                    value=False
                )
                
                stereo_checkbox = gr.Checkbox(
                    label="🔊 모노 → 스테레오 변환 (한쪽 소리를 양쪽으로)",
                    value=False,
                    info="한쪽 이어폰/스피커에서만 들리는 소리를 양쪽에서 들리게 만듭니다"
                )
                
                process_btn = gr.Button(
                    "🚀 영상 처리 시작",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column():
                status_output = gr.Textbox(
                    label="📊 처리 상태",
                    lines=8,
                    interactive=False
                )
                
                video_output = gr.File(
                    label="📥 처리된 영상 다운로드",
                    interactive=False
                )
        
        # 예제 섹션
        gr.Markdown("""
        ## 💡 사용 팁
        - **최적 성능**: 48kHz 샘플링 레이트의 오디오가 포함된 영상을 사용하세요
        - **처리 시간**: 영상 길이에 따라 몇 분에서 수십 분이 소요될 수 있습니다
        - **GPU 가속**: CUDA가 설치된 환경에서는 GPU를 자동으로 활용합니다
        
        ## ⚠️ 주의사항
        - FFmpeg가 시스템에 설치되어 있어야 합니다
        - DeepFilterNet 모델 다운로드에 시간이 걸릴 수 있습니다
        - 대용량 파일 처리 시 충분한 디스크 공간을 확보해주세요
        """)
        
        # 이벤트 핸들러
        process_btn.click(
            fn=process_video,
            inputs=[video_input, postfilter_checkbox, stereo_checkbox],
            outputs=[video_output, status_output],
            show_progress=True
        )
        
        return demo

# 의존성 확인 함수
def check_dependencies():
    """필요한 의존성 확인"""
    dependencies = {
        'ffmpeg': 'FFmpeg (영상/오디오 처리)',
        'deepfilternet': 'DeepFilterNet (음성 향상)',
        'torch': 'PyTorch (딥러닝)',
        'torchaudio': 'TorchAudio (오디오 처리)'
    }
    
    missing = []
    
    # FFmpeg 확인
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append('ffmpeg')
    
    # Python 패키지 확인
    try:
        import torch
        import torchaudio
        from df import enhance, init_df
    except ImportError as e:
        missing.append(str(e))
    
    if missing:
        print("❌ 다음 의존성이 누락되었습니다:")
        for dep in missing:
            print(f"   - {dep}")
        print("\n설치 방법:")
        print("pip install torch torchaudio deepfilternet")
        print("FFmpeg: https://ffmpeg.org/download.html")
        return False
    
    print("✅ 모든 의존성이 설치되어 있습니다.")
    return True

if __name__ == "__main__":
    print("🎬 Video Audio Enhancement with DeepFilterNet")
    print("=" * 50)
    
    # 의존성 확인
    if not check_dependencies():
        exit(1)
    
    # Gradio 앱 실행
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )