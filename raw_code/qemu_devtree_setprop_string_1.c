int qemu_devtree_setprop_string(void *fdt, const char *node_path,

                                const char *property, const char *string)

{

    int offset;



    offset = fdt_path_offset(fdt, node_path);

    if (offset < 0)

        return offset;



    return fdt_setprop_string(fdt, offset, property, string);

}
