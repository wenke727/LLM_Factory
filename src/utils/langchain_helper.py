from IPython.display import Image, display

def display_graph(graph):
    """使用Mermaid来可视化图。

    Args:
    graph: 一个图对象，应该有一个 `get_graph()` 方法，该方法返回一个可以绘制为Mermaid图的对象。
    """
    try:
        # 尝试生成和显示Mermaid图
        mermaid_graph = graph.get_graph()  # 获取可以用Mermaid绘制的图对象
        png_image = mermaid_graph.draw_mermaid_png()  # 生成PNG图片
        display(Image(png_image))
    except Exception as e:
        # 如果有任何错误，打印错误信息
        print("无法生成图像:", e)

# 假设你有一个图对象，你可以这样调用函数：
# display_graph(your_graph_object)
