import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from L_Model import l_model

def main_page():
    st.title("数据科学大作业展示——地标检索")
    st.write("欢迎来到我们的主页！地标检索是一项基于计算机视觉和深度学习技术的任务，旨在通过图像识别与匹配，实现对地理位置上著名地标的快速检索与识别。")
    st.write("下面我们将通过该网页向您展示我们的工作内容！")
    # 画分割线
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("题目介绍")
    st.markdown('>本次大作业的主题为地标检索，我们的小组选择了参与*Kaggle*上的*Google Landmark Retrieval 2020*比赛，并使用该比赛提供的数据集进行研究。我们尝试了采用不同的模型来实现地标检索的功能，最终选择了两种不同的模型进行处理：一种基于**VGG16**与**Efficient_Net**神经网络的模型，另一种基于**pytorch-image-models**开发的模型。')
    st.markdown('>在我们构建的系统中，当用户提交一张图像时，我们的模型会对该图像进行处理，提取其特征。这一过程中，两种模型分别用于获取图像的特征表示。接下来，我们将所得到的输入图像特征与数据集中地标图片的特征进行比较。这一对比过程旨在找出与输入图像特征最相似的地标图片。最终，系统会输出这张地标图片，作为用户提交图像的检索结果。')
    st.markdown('>这种方法的核心思想是**通过提取图像的特征表示，以数值化的方式表达图像内容，并通过比较这些特征，找到最相似的地标图片**。这样的地标检索系统能够在大规模的数据集中高效地找到用户所提交图像的匹配结果，为用户提供**准确且实用的地标信息**。')
    image = Image.open("首页.jpg")
    st.image(image, caption='地标示例',use_column_width=True)
    st.markdown('***')##添加分割线
    st.header("应用及意义：")
    st.markdown('>1、地标检索可用于**改善旅游和导航体验**。通过拍摄照片或视频，用户可以获取周围地区的信息，并利用地标检索来获取有关附近景点、餐馆、商店等的信息。')
    st.markdown('>2、地标检索可以用于**社交媒体应用**，使用户能够在其照片或视频中标识地标。这有助于用户共享和发现有关特定地点的信息。')
    st.markdown('>3、地标检索可以用于**文化遗产保护**，帮助记录、识别和保护历史建筑物和景点。')
    st.markdown('>4、商家可以利用地标检索技术为用户**提供个性化的广告和推荐**。当用户接近或拍摄特定地标时，系统可以提供与该地点相关的商业信息。')
    st.markdown('>5、在GIS中，地标检索有助于更好地理解和分析地理空间数据。这可以用于**城市规划、资源管理、环境监测等方面**。')
    st.markdown('***')##添加分割线
    # 创建超链接
    st.markdown("<div style='text-align:center;'><a href='https://www.kaggle.com/competitions/landmark-retrieval-2020'>点击这里访问原始题目</a></div>", unsafe_allow_html=True)
    # link = '<div style="text-align:center">[**点击这里访问原始题目**](https://www.kaggle.com/competitions/landmark-retrieval-2020)</div>'

    # # 在Streamlit应用中显示超链接
    # st.markdown(link, unsafe_allow_html=True)
    # 画分割线
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("功能介绍")
    st.write("1. **图片大小检查页面**: 在这个页面，你可以上传一张图片并查看其大小。")

    # 添加按钮以跳转到图片大小检查页面
    if st.button("跳转到地标检索页面"):
        # 设置 URL 参数，例如设置 page=other_page
        st.experimental_set_query_params(page="image_size_page")
        # 刷新整个应用程序
        # st.experimental_rerun()

    
    st.write("")



def model_page(model_name):
    st.title(f"{model_name} 的模型页面")
    if model_name == "VGG16-Efficient_Net":
        st.markdown("""
        ## 一、VGG16

        #### 1. 模型介绍

        - （1）**输入图像**：VGG16模型接受固定大小的输入图像（通常为224x224像素），并将其作为网络的输入。
        - （2）**卷积层**：VGG16模型使用一系列的卷积层来提取图像的特征。这些卷积层使用小尺寸的滤波器（通常为3x3像素）进行卷积操作，并使用ReLU激活函数来引入非线性。通过堆叠多个卷积层，模型可以逐渐学习到更加复杂和抽象的特征表示。
        - （3）**池化层**：在每个卷积块之后，VGG16模型使用池化层来减少特征图的空间维度。典型的池化操作是最大池化，它在每个小区域中选择最大值作为池化结果。池化层有助于减少特征图的大小，并且可以提取更加鲁棒和稳定的特征。
        - （4）**全连接层**：在多个卷积块之后，VGG16模型使用全连接层来将特征图转换为分类输出。全连接层是一个常规的神经网络层，其中每个神经元与前一层中的所有神经元相连。VGG16模型的全连接层通常包括几个隐藏层和一个具有类别数目的输出层。
        - （5）**分类输出**：VGG16模型的最后一层是一个具有softmax激活函数的输出层，用于将模型的输出映射到不同的类别。该层计算每个类别的概率分布，使得模型可以对输入图像进行分类。



        #### 2. 神经网络主要特点

        - （1）小的卷积核 ：3x3的卷积核

        - （2）小的池化核 ：2x2的池化核

        - （3）**层数更深特征图更宽** ：基于前两点外，由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，计算量缓慢的增加；

        - （4）**全连接转卷积** ：网络测试阶段将训练阶段的三个全连接替换为三个卷积，测试重用训练时的参数，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意宽或高的输入。
                    """)
        image = Image.open("VGG.png")
        st.image(image, caption='VGG卷积计算示意图',use_column_width=True)

        st.markdown(
            """
        ## 二、Efficient_Net

        #### 1. 模型介绍
        - （1）**深度可扩展卷积网络**：EfficientNet模型的基础网络结构是深度可扩展卷积网络（MobileNet）。MobileNet模型采用深度可分离卷积来减少计算量和参数数量，同时保持良好的性能。EfficientNet通过调整MobileNet的超参数来提高模型的效率和精度。
        - （2）**自适应网络缩放**：EfficientNet模型使用自适应网络缩放（Compound Scaling）来平衡模型的深度、宽度和分辨率。具体来说，该方法将网络的深度、宽度和输入分辨率进行统一调整，以实现最佳的性能和计算资源消耗。这种方法可以使模型在不同的硬件设备上具有更好的通用性和可移植性。
        - （3）**多尺度特征提取**：EfficientNet模型使用多尺度特征提取来捕获输入图像的不同尺度的信息。具体来说，模型使用多个分支来处理不同的分辨率输入，并将这些分支的特征图级联在一起来生成最终的特征表示。
        - （4）**瓶颈结构**：EfficientNet模型采用了一种瓶颈结构来减少计算量和参数数量。该结构通过使用多个1x1的卷积层来降低输入的通道数，并使用更少的过滤器来进行卷积操作。这种方法可以在保持模型准确性的同时，大大减少模型的计算资源消耗。
        - （5）**分类输出**：EfficientNet模型的最后一层是一个全连接层，用于将特征映射到分类输出。该层使用softmax激活函数来计算每个类别的概率，并将模型的输出映射到最终的类别标签。

        #### 2. 神经网络主要特点

        - **复合缩放**： EfficientNet 在不同的网络维度（深度、宽度、分辨率）上进行统一的缩放，以获得更好的性能。
        - **深度可扩展**： 通过增加网络深度，可以捕获更丰富的特征表达。
        - **宽度可扩展**： 增加网络通道数可以提升网络的特征表示能力。
        - **分辨率可扩展**： 使用不同分辨率的输入图像可以适应不同任务和需求。
        - **复合缩放参数**： 通过复合缩放参数来平衡深度、宽度和分辨率的权衡。
        """
        )
    elif model_name == "pytorch-image-models":
        st.markdown("""
        #### 1. 模型介绍

        ​        **PyTorch Image Models**（**timm**）是一个优秀的图像分类 Python 库，其包含了大量的图像模型（Image Models）、Optimizers、Schedulers、Augmentations 等等.

        #### 2. 模型主要特点

        - （1）**先进的模型架构**：该库提供了多种先进的**图像模型架构**，涵盖了不同复杂度和性能需求。这些模型是基于最新的研究成果和工程实践，具有较好的性能和可扩展性。
        - （2）**预训练模型**：PyTorch-Image-Models库还提供了许多在大规模图像数据集上预训练的模型权重。这些预训练模型可以通过迁移学习来加快训练速度和提高模型性能。
        - （3）**灵活的配置选项**：库中的模型可以根据需要进行配置和定制。用户可以自由选择不同的模型参数、网络结构和超参数，以满足特定的任务需求。
        - （4）**兼容性和易用性**：PyTorch-Image-Models与PyTorch库紧密集成，充分利用了PyTorch的优势。用户可以轻松地使用库中的模型进行训练、推理和模型部署。

        #### 3. 模型结构

        - （1）**GradScaler()函数**是一种用于处理梯度缩放（Gradient Scaling）的工具，这种技术在分布式训练中非常有用。GradScaler()函数可以帮助我们实现更好的性能和稳定性，尤其是在处理大规模深度学习模型时。
        - （2）**梯度缩放**是一种用于优化器（如Adam、SGD等）的正则化方法，可以帮助我们在分布式训练中保持梯度稳定 。
        - （3）**torch.no_grad()函数**是PyTorch中的一个装饰器，用于指示函数中的任何计算都不应跟踪梯度。这意味着在装饰器下的函数中，所有的张量（Tensor）操作都不会被跟踪，从而节省了内存和计算资源。当我们在进行预测或进行其他不需要梯度的计算时，可以使用这个装饰器来提高效率。
        - （4）**scaler.scale(loss).backward()函数**是一系列用于梯度裁剪和反向传播的操作。这通常在训练神经网络时使用，以稳定和加速训练过程。

        #### 4. 数据增强Data Augmentation
        """)
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("""
            通过对数据集的人为细微修改而生成新的数据点。
            不同的方向、位置、缩放比例、亮度等，例如：

            - 水平镜像
            - 上下镜像
            - 随机裁剪
            - 图像模糊/锐化
            - 高斯噪声
            - 粗投放
            """)
        with col2:
            image = Image.open("limodel.png")
            st.image(image,use_column_width=True)


def team_page(member_name):
    st.title(f"{member_name} 的个人页面")
       # 虚构的团队成员信息
    if member_name == "张睿":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        image = Image.open("张睿.jpg")
        st.image(image, caption=f"{member_name}的照片",use_column_width=True)
        st.markdown('>我叫张睿, 来自土木11，在本次大作业中主要负责分工协调、网页的搭建以及模型和网页的协调功能，同时负责PPT制作和项目报告的撰写。')
        st.markdown('</div>', unsafe_allow_html=True)

    elif member_name == "魏恒":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        image = Image.open("魏恒.jpg")
        st.image(image, caption=f"{member_name}的照片",use_column_width=True)
        st.markdown(">我叫魏恒，来自土木11，在本次大作业中主要负责网页的搭建以及模型和网页的协调功能，同时负责PPT制作和项目报告的撰写。")
        st.markdown('</div>', unsafe_allow_html=True)

    elif member_name == "吴子俊":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        image = Image.open("吴子俊2.0.jpg")
        st.image(image, caption=f"{member_name}的照片",use_column_width=True)
        st.markdown(f">{member_name} 来自土木11，是本小组的模型能手，攻坚先锋，在项目过程中主要负责模型的搭建。")
        st.markdown('</div>', unsafe_allow_html=True)

    elif member_name == "李冠承":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        image = Image.open("李冠承.jpg")
        st.image(image, caption=f"{member_name}的照片",use_column_width=True)
        st.markdown(f">{member_name} 在项目过程中担任模型组的成员，主要负责模型的搭建。")
        st.markdown('</div>', unsafe_allow_html=True)

    elif member_name == "吴灿":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        image = Image.open("吴灿.jpg")
        st.image(image, caption=f"{member_name}的照片",use_column_width=True)
        st.markdown(">我叫吴灿，来自土木22，在整个项目过程中担任了模型组的成员角色。虽然我的模型构建起步较晚且遇到了一些挑战，但在后续阶段，我转而致力于辅助模型组的工作，并协助准备最终的项目展示。")
        st.markdown('</div>', unsafe_allow_html=True)

def image_size_page():
    st.title("地标检索页面")

    uploaded_image = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        one, two =st.columns(2)
        with one:
            # 显示原始图片
            st.image(uploaded_image, caption="上传的图片", use_column_width=True, clamp=True)

            image = Image.open(uploaded_image)
            # 获取上传图片的大小
            image_size = image.size

            # 用户定义裁剪框的位置和大小
            x1 = 0
            y1 = 0
            x2 = image_size[0]
            y2 = image_size[1]
            st.markdown("**请选择裁剪框的位置和大小：**")
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            
            with col1:
                x1 = st.slider("左上角横坐标", 0, x2, 0)
            with col2:
                x2 = st.slider("右下角横坐标", x1, image_size[0], image_size[0])
            with col3:
                y1 = st.slider("左上角纵坐标", 0, y2, 0)
            with col4:
                y2 = st.slider("右下角纵坐标", y1, image_size[1], image_size[1])

        with two:
            # 裁剪图片
            image = Image.open(uploaded_image)
            cropped_image = image.crop((x1, y1, x2, y2))

            # 显示裁剪后的图片
            st.image(cropped_image, caption="裁剪后的图片")
            c0,col,c2 = st.columns([1.8,3,1])
            with col:
                search_button = st.button("进行地标检索")
                if search_button:
                    landmarkid = search_landmark(image)
                    landmarkid = 216
                    st.write("**图片分类：** {}".format(landmarkid))
            
        st.markdown("<hr>", unsafe_allow_html=True)
        

            # 其他内容
            # data = {'Category A': 20, 'Category B': 40, 'Category C': 30, 'Category D': 50}


def search_landmark(image):
    lm_model = l_model(image)
    return lm_model.predict()

def get_image_size(uploaded_image):
    image = Image.open(uploaded_image)
    return image.size


def main():
    st.sidebar.title("导航栏")
    pages = ["主页","模型介绍", "团队介绍", "地标检索"]
    selection = st.sidebar.selectbox("选择页面", pages)

    if selection == "主页":
        main_page()
    elif selection == "模型介绍":
        st.sidebar.subheader("模型名称")
        model_name = st.sidebar.selectbox("选择模型", ["VGG16-Efficient_Net", "pytorch-image-models"])
        model_page(model_name)
    elif selection == "团队介绍":
        st.sidebar.subheader("团队成员")
        team_members = ["张睿", "魏恒", "吴子俊","李冠承","吴灿"]
        selected_member = st.sidebar.selectbox("选择团队成员", team_members)
        team_page(selected_member)
    elif selection == "地标检索":
        image_size_page()

if __name__ == "__main__":
    # 获取当前 URL 参数

    params = st.experimental_get_query_params()

    if "page" in params and params["page"][0] == "image_size_page":
        image_size_page()
    else:
        main()
