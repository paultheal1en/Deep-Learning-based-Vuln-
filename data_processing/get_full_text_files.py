import os
import json

def main(project, split_type):
    """
    Hàm chính để xử lý các tập tin trong thư mục chỉ định và lưu chúng vào định dạng JSON.
    
    Tham số:
    - project (str): Tên của thư mục dự án.
    - split_type (str): Loại dữ liệu (ví dụ: "train", "valid", hoặc "test").
    """
    # Khởi tạo danh sách để lưu dữ liệu từ các tập tin
    lst = []
    
    # Duyệt qua từng tập tin trong thư mục chứa mã nguồn của dự án với loại dữ liệu đã chỉ định
    for file_name in os.listdir(f'/data3/dlvp_local_data/dataset_merged/{project}/all_{split_type}/code'):
        # Tạo một từ điển để lưu thông tin của từng tập tin
        dic = {}
        
        # Kiểm tra nếu tập tin có đuôi "1.c"
        if file_name.endswith('1.c'):
            # Mở tập tin và đọc nội dung mã nguồn
            with open(f'/data3/dlvp_local_data/dataset_merged/{project}/all_{split_type}/code/'+file_name, 'r') as f:
                code = f.read()
            # Lưu mã nguồn vào từ điển với nhãn "1" và tên tập tin
            dic['code'] = code
            dic['label'] = 1
            dic['file_name'] = file_name
            # Thêm từ điển vào danh sách
            lst.append(dic)
        
        # Kiểm tra nếu tập tin có đuôi "0.c"
        if file_name.endswith('0.c'):
            # Mở tập tin và đọc nội dung mã nguồn
            with open(f'/data3/dlvp_local_data/dataset_merged/{project}/all_{split_type}/code/'+file_name, 'r') as f:
                code = f.read()
            # Lưu mã nguồn vào từ điển với nhãn "0" và tên tập tin
            dic['code'] = code
            dic['label'] = 0
            dic['file_name'] = file_name
            # Thêm từ điển vào danh sách
            lst.append(dic)

    # Lưu danh sách các từ điển vào một file JSON
    with open(f'/data3/dlvp_local_data/dataset_merged/{project}/all_{split_type}/{project}_{split_type}_cfg_full_text_files.json', 'w+') as f:
        json.dump(lst, f)
        
# Phần code chạy chương trình chính
if __name__ == "__main__":
    # Đặt tên dự án
    PROJECT = "new_six_datasets"
    # Duyệt qua từng loại dữ liệu ("train", "valid", "test") và gọi hàm `main`
    for SPLIT_TYPE in ["train", "valid", "test"]:
        main(PROJECT, SPLIT_TYPE)
