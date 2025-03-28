import docx
import nbformat as nbf
import os
import sys

def convert_word_to_ipynb(word_file, output_file):
    try:
        print(f"开始读取Word文档: {word_file}")
        # 读取Word文档
        doc = docx.Document(word_file)
        
        print("创建新的notebook")
        # 创建新的notebook
        nb = nbf.v4.new_notebook()
        
        # 遍历Word文档中的段落
        current_cell_content = []
        cell_count = 0
        
        print("开始处理文档内容...")
        for para in doc.paragraphs:
            if para.text.strip():
                current_cell_content.append(para.text)
            elif current_cell_content:
                # 当遇到空段落时，创建一个新的cell
                cell_content = '\n'.join(current_cell_content)
                nb['cells'].append(nbf.v4.new_markdown_cell(cell_content))
                current_cell_content = []
                cell_count += 1
        
        # 处理最后一个cell
        if current_cell_content:
            cell_content = '\n'.join(current_cell_content)
            nb['cells'].append(nbf.v4.new_markdown_cell(cell_content))
            cell_count += 1
        
        print(f"创建了 {cell_count} 个单元格")
        
        # 保存notebook
        print(f"正在保存notebook到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
            
        print("转换完成！")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    word_file = '3.62 MCP快速入门实战.docx'
    output_file = 'MCP快速入门实战.ipynb'
    
    # 检查文件是否存在
    if not os.path.exists(word_file):
        print(f"错误: 找不到Word文档 {word_file}")
        sys.exit(1)
        
    convert_word_to_ipynb(word_file, output_file) 