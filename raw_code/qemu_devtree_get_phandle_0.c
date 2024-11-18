uint32_t qemu_devtree_get_phandle(void *fdt, const char *path)

{

    uint32_t r;



    r = fdt_get_phandle(fdt, findnode_nofail(fdt, path));

    if (r <= 0) {

        fprintf(stderr, "%s: Couldn't get phandle for %s: %s\n", __func__,

                path, fdt_strerror(r));

        exit(1);

    }



    return r;

}
