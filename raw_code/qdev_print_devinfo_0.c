static int qdev_print_devinfo(DeviceInfo *info, char *dest, int len)

{

    int pos = 0;



    pos += snprintf(dest+pos, len-pos, "name \"%s\", bus %s",

                    info->name, info->bus_info->name);

    if (info->alias)

        pos += snprintf(dest+pos, len-pos, ", alias \"%s\"", info->alias);

    if (info->desc)

        pos += snprintf(dest+pos, len-pos, ", desc \"%s\"", info->desc);

    if (info->no_user)

        pos += snprintf(dest+pos, len-pos, ", no-user");

    return pos;

}
