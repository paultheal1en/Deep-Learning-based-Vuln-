char* qdev_get_fw_dev_path(DeviceState *dev)

{

    char path[128];

    int l;



    l = qdev_get_fw_dev_path_helper(dev, path, 128);



    path[l-1] = '\0';



    return strdup(path);

}
