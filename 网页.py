import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

def main_page():
    st.title("数据科学大作业展示——地标检索")
    st.write("欢迎来到我们的主页！地标检索是一项基于计算机视觉和深度学习技术的任务，旨在通过图像识别与匹配，实现对地理位置上著名地标的快速检索与识别。")
    st.write("下面我们将通过该网页向您展示我们的工作内容！")
    # 画分割线
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("题目介绍")
    st.markdown('>本次大作业的主题为地标检索，我们的小组选择了参与Kaggle上的*Google Landmark Retrieval 2020*比赛，并使用该比赛提供的数据集进行研究。我们尝试了采用不同的模型来实现地标检索的功能，最终选择了两种不同的模型进行处理：一种**基于VGG50**，另一种**基于DELF**。')
    st.markdown('>在我们构建的系统中，当用户提交一张图像时，我们的模型会对该图像进行处理，提取其特征。这一过程中，基于**VGG50**的模型和基于**DELF**的模型分别用于获取图像的特征表示。接下来，我们将所得到的输入图像特征与数据集中地标图片的特征进行比较。这一对比过程旨在找出与输入图像特征最相似的地标图片。最终，系统会输出这张地标图片，作为用户提交图像的检索结果。')
    st.markdown('>这种方法的核心思想是**通过提取图像的特征表示，以数值化的方式表达图像内容，并通过比较这些特征，找到最相似的地标图片**。这样的地标检索系统能够在大规模的数据集中高效地找到用户所提交图像的匹配结果，为用户提供**准确且实用的地标信息**。')
    image = Image.open("photo\\首页.jpg")
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
    link = '[**点击这里访问原始题目**](https://www.kaggle.com/c/landmark-retrieval-2021)'

    # 在Streamlit应用中显示超链接
    st.markdown(link, unsafe_allow_html=True)
    # 画分割线
    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("功能介绍")
    st.write("1. **图片大小检查页面**: 在这个页面，你可以上传一张图片并查看其大小。")

    # 添加按钮以跳转到图片大小检查页面
    if st.button("跳转到地标检索页面"):
        # 设置 URL 参数，例如设置 page=other_page
        st.experimental_set_query_params(page="image_size_page")
        # 刷新整个应用程序
        st.experimental_rerun()

    
    st.write("")



def model_page(model_name):
    st.title(f"{model_name} 的模型页面")
    if model_name == "ResNet":
        st.markdown()
    elif model_name == "AlexNet":
        st.markdown()
    elif model_name == "VGG":
        st.markdown()



def team_page(member_name):
    st.title(f"{member_name} 的个人页面")
       # 虚构的团队成员信息
    if member_name == "张睿":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.image("photo\张睿.jpg", caption=f"{member_name} 的照片", use_column_width=True)
        st.write(f"{member_name} 是我们团队的创始人之一。他喜欢编程和创新。")
        st.markdown('</div>', unsafe_allow_html=True)
    elif member_name == "魏恒":
        image_path = r"photo\魏恒.jpg"
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.image(image_path, caption=f"{member_name} 的照片", use_column_width=True)
        st.write(f"{member_name} 是我们团队的设计师。他擅长图形设计和用户体验。")
        st.markdown('</div>', unsafe_allow_html=True)
    elif member_name == "吴子俊":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.image("photo\吴子俊1.0.jpg", caption=f"{member_name} 的照片", use_column_width=True)
        st.write(f"{member_name} 是我们团队的开发者。他对新技术充满热情。")
        st.markdown('</div>', unsafe_allow_html=True)
    elif member_name == "李冠承":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.image("photo\李冠承.jpg", caption=f"{member_name} 的照片", use_column_width=True)
        st.write(f"{member_name} 是我们团队的设计师。他擅长图形设计和用户体验。")
        st.markdown('</div>', unsafe_allow_html=True)
    elif member_name == "吴灿":
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        st.image("photo\吴灿.jpg", caption=f"{member_name} 的照片", use_column_width=True)
        st.write(f"{member_name} 是我们团队的设计师。他擅长图形设计和用户体验。")
        st.markdown('</div>', unsafe_allow_html=True)

def image_size_page():
    st.title("地标检索页面")

    # if uploaded_image is not None and calculate_button:
    # 用户上传图片
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
            
        st.markdown("<hr>", unsafe_allow_html=True)
        if search_button:
            width, height = search_landmark(image)
            st.write("")
            st.write("**图片大小：**{} x {}".format(width, height))

            # 其他内容
            data = {'Category A': 20, 'Category B': 40, 'Category C': 30, 'Category D': 50}


def search_landmark(image):
    return image.size

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
        model_name = st.sidebar.selectbox("选择模型", ["模型1", "模型2"])
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

