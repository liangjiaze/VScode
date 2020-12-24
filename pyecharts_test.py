import pyecharts

print(pyecharts.__version__)
# pyecharts官方文档：https://pyecharts.org/#/zh-cn/quickstart

# --------------------------------------------------添加第一个柱状图----------------------------------------------------
from pyecharts.charts import Bar
from pyecharts import options as opts
# 设置图形主题，内置主题类型可查看 pyecharts.globals.ThemeType
from pyecharts.globals import ThemeType

filepath = r'D:\jade\Desktop\myEcharts0.html'
bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
bar.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
bar.add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
bar.add_yaxis("商家B", [15, 6, 45, 20, 35, 66])
bar.set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"))
bar.render(filepath)





