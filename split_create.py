import os


if __name__ == "__main__":
    data_path = "/media/terryxu/My Passport/kitti/kiti_data_jpg"
    # print(os.listdir("/media/terryxu/My Passport/kitti/kiti_data_jpg"))
    filenames = os.listdir(data_path)
    for filename in filenames:
        foldernames=os.listdir(data_path+"/"+filename)
        # print(os.listdir(data_path+"/"+filename))
        for name in foldernames:
            with open('test.txt', 'a') as f:
                f.write(filename+"/"+name)
                f.write('\n')

