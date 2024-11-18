int qemu_devtree_add_subnode(void *fdt, const char *name)

{

    int offset;

    char *dupname = g_strdup(name);

    char *basename = strrchr(dupname, '/');

    int retval;



    if (!basename) {

        return -1;

    }



    basename[0] = '\0';

    basename++;



    offset = fdt_path_offset(fdt, dupname);

    if (offset < 0) {

        return offset;

    }



    retval = fdt_add_subnode(fdt, offset, basename);

    g_free(dupname);

    return retval;

}
