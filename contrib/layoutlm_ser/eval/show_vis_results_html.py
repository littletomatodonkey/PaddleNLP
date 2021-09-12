import os


def draw_cases():
    imgs = os.listdir("./output_res_v1/")
    html_path = "./results/show_infer_with_anno_and_e2e.html"
    with open(html_path, 'w') as html:
        html.write('<html>\n<body>\n')
        html.write('<table border="1">\n')
        html.write(
            "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\" />"
        )
        for img_name in imgs:
            img_path = "./output_res_v1/{}".format(img_name)
            img_e2e_path = "./output_e2e_v1/{}".format(img_name)

            html.write("<tr>\n")
            html.write('<td>%s</td>' % (img_name))
            html.write('<td><img src="../%s" width=640></td>' % (img_path))

            html.write('<td><img src="../%s" width=640></td>' % (img_e2e_path))
            html.write("</tr>\n")
        html.write('</table>\n')
        html.write('</html>\n</body>\n')
    print("ok")
    return


if __name__ == "__main__":
    draw_cases()
