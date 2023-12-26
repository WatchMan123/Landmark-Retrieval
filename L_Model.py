import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from kaggle.api.kaggle_api_extended import KaggleApi
import timm


class l_model():
    def __init__(self, image) ->None :
        self.image = image

    def predict(self):
        
        # 以ResNet为例，你需要替换为实际竞赛中使用的模型
        model = timm.create_model('tf_mobilenetv3_small_100',pretrained=False)

        # 加载.pth文件
        checkpoint = torch.load('final_model.pth')
        # 获取模型定义中的键
        model_keys = set(model.state_dict().keys())

        # 获取权重文件中的键
        checkpoint_keys = set(checkpoint.keys())

        # 查找缺失的键
        missing_keys = model_keys - checkpoint_keys

        # 打印缺失的键
        print(f"Missing key(s) in state_dict: {missing_keys}")

        # 根据缺失的键，逐个添加到模型权重中
        for key in missing_keys:
            if key in checkpoint:
                model.state_dict()[key].copy_(checkpoint[key])
            else:
                print(f"Error: Key '{key}' not found in checkpoint.")
        # 从 checkpoint 中分别加载主干部分和头部部分的权重
        # model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        # model.head.load_state_dict(checkpoint['head_state_dict'])
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        # 预处理图像
        CONFIG = dict(
            seed = 42,
            model_name = 'tf_mobilenetv3_small_100',
            train_batch_size = 384,
            valid_batch_size = 768,
            img_size = 224,
            epochs = 3,
            learning_rate = 5e-4,
            scheduler = None,
            # min_lr = 1e-6,
            # T_max = 20,
            # T_0 = 25,
            # warmup_epochs = 0,
            weight_decay = 1e-6,
            n_accumulate = 1,
            n_fold = 5,
            num_classes = 81313,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            competition = 'GOOGL',
            _wandb_kernel = 'deb'
        )

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_image = transform(self.image).unsqueeze(0)  # 添加 batch 维度

        # 如果你的模型支持 GPU，将输入数据移到 GPU 上
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_image = input_image.to(device)

        # 进行预测
        with torch.no_grad():
            output = model(input_image)

        # 处理预测结果
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()

        # dataset_path = KaggleDatasets().get_gcs_path('landmark-retrieval-2020')

        return predicted_class
