import os
import sys
import torch
import numpy as np
from PIL import Image
import time
from pathlib import Path
from typing import Optional
import torchvision.transforms as transforms

# UNSB 모듈들 import
sys.path.append('UNSB')
from UNSB.models import create_model
from UNSB.options.test_options import TestOptions

class SchrodingerBridgeProcessor:
    def __init__(self, 
                 device: str = "cuda:0",
                 checkpoints_dir: str = "models/checkpoints",
                 model_name: str = "225_net_G_B2A",  # 체크포인트 접두어
                 epoch: str = "latest"):  # 에포크 번호
        """슈뢰딩거 브릿지 프로세서 초기화"""
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        self.model_name = model_name
        self.epoch = epoch
        self.model = None
        
        print("SchrodingerBridge 프로세서 초기화 중...")
        self._load_unsb_model()
        print("SchrodingerBridge 프로세서 초기화 완료")
    
    def _load_unsb_model(self):
        """UNSB 모델 로드"""
        try:
            # args 없이 기본 초기화 후 수동 설정
            opt = TestOptions().parse()
            
            # 필요한 속성들 수동으로 설정
            opt.name = self.model_name
            opt.checkpoints_dir = self.checkpoints_dir
            opt.epoch = self.epoch
            opt.model = 'sb'
            opt.phase = 'test'
            opt.batch_size = 1
            opt.num_threads = 1
            opt.serial_batches = True
            opt.no_flip = True
            opt.eval = True
            opt.gpu_ids = [0]
            opt.dataroot = './dummy'
            opt.load_size = 256
            opt.crop_size = 256
            opt.direction = 'BtoA'
            opt.dataset_mode = 'unaligned'
            
            # 모델 생성 및 로드
            self.model = create_model(opt)
            self.opt = opt
            
            # 더미 데이터로 초기화
            dummy_tensor = torch.randn(1, 3, 256, 256)
            dummy_data = {
                'A': dummy_tensor,
                'B': dummy_tensor,  # B 데이터도 추가
                'A_paths': ['dummy'],
                'B_paths': ['dummy']  # B 경로도 추가
            }
            
            self.model.data_dependent_initialize(dummy_data, dummy_data)
            self.model.setup(opt)
            self.model.eval()
            
            print(f"UNSB 모델 로딩 완료: {self.model_name}, epoch {self.epoch}")
            
        except Exception as e:
            print(f"UNSB 모델 로딩 실패: {e}")
            print("스택 트레이스:")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL Image를 모델 입력용 Tensor로 변환"""
        # 256x256으로 리사이즈
        pil_image = pil_image.resize((256, 256), Image.BICUBIC)
        
        # 정규화: [0, 1] -> [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        tensor = transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Tensor를 PIL Image로 변환"""
        # 배치 차원 제거
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # [-1, 1] -> [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        
        # CPU로 이동 후 numpy 변환
        if tensor.is_cuda:
            tensor = tensor.cpu()
        numpy_array = tensor.detach().numpy()
        numpy_array = np.transpose(numpy_array, (1, 2, 0))
        numpy_array = (numpy_array * 255).astype(np.uint8)
        
        return Image.fromarray(numpy_array)
    
    def real_to_illustration(self, image_path: str, output_dir: str = "temp") -> str:
        """실제 이미지 → 일러스트화"""
        if self.model is None:
            return self._placeholder_process(image_path, output_dir, "real_to_illust")
        
        try:
            # 입력 이미지 로드
            input_image = Image.open(image_path).convert("RGB")
            input_tensor = self._pil_to_tensor(input_image)
            
            # 출력 경로 생성
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"real_to_illust_{int(time.time())}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 모델 inference
            with torch.no_grad():
                # 입력 데이터 설정
                data = {
                    'A': input_tensor,
                    'B': input_tensor,  
                    'A_paths': [image_path],
                    'B_paths': [image_path]
                }
                
                self.model.set_input(data, data)
                self.model.test()
                
                # 결과 가져오기
                visuals = self.model.get_current_visuals()
                if 'fake_5' in visuals:
                    output_tensor = visuals['fake_5']  # 최종 변환 결과
                elif 'fake_B' in visuals:
                    output_tensor = visuals['fake_B']
                elif 'fake' in visuals:
                    output_tensor = visuals['fake']
                else:
                    print(f"사용 가능한 결과 키들: {list(visuals.keys())}")
                    # fake_로 시작하는 키 중 가장 큰 번호 선택
                    fake_keys = [k for k in visuals.keys() if k.startswith('fake_')]
                    if fake_keys:
                        # fake_1, fake_2, ... 중 마지막 선택
                        last_fake = max(fake_keys, key=lambda x: int(x.split('_')[1]))
                        output_tensor = visuals[last_fake]
                    else:
                        output_tensor = list(visuals.values())[0]
                                
                # 결과 저장
                output_image = self._tensor_to_pil(output_tensor)
                output_image.save(output_path)
            
            print(f"실제→일러스트 변환 완료: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"실제→일러스트 변환 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._placeholder_process(image_path, output_dir, "real_to_illust")
    
    def illustration_to_real(self, image_path: str, output_dir: str = "temp") -> str:
        """일러스트 → 실제 이미지화 (B2A 방향)"""
        # 현재는 같은 모델 사용, 필요시 B2A 모델로 교체
        return self.real_to_illustration(image_path, output_dir)
    
    def _placeholder_process(self, image_path: str, output_dir: str, mode: str) -> str:
        """모델 실패시 placeholder 처리"""
        try:
            input_image = Image.open(image_path).convert("RGB")
            
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{mode}_{int(time.time())}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            input_image.save(output_path)
            print(f"[PLACEHOLDER] {mode}: {image_path} → {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Placeholder 처리 실패: {e}")
            raise