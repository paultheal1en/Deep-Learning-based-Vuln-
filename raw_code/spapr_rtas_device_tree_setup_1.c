int spapr_rtas_device_tree_setup(void *fdt, hwaddr rtas_addr,

                                 hwaddr rtas_size)

{

    int ret;

    int i;



    ret = fdt_add_mem_rsv(fdt, rtas_addr, rtas_size);

    if (ret < 0) {

        fprintf(stderr, "Couldn't add RTAS reserve entry: %s\n",

                fdt_strerror(ret));

        return ret;

    }



    ret = qemu_fdt_setprop_cell(fdt, "/rtas", "linux,rtas-base",

                                rtas_addr);

    if (ret < 0) {

        fprintf(stderr, "Couldn't add linux,rtas-base property: %s\n",

                fdt_strerror(ret));

        return ret;

    }



    ret = qemu_fdt_setprop_cell(fdt, "/rtas", "linux,rtas-entry",

                                rtas_addr);

    if (ret < 0) {

        fprintf(stderr, "Couldn't add linux,rtas-entry property: %s\n",

                fdt_strerror(ret));

        return ret;

    }



    ret = qemu_fdt_setprop_cell(fdt, "/rtas", "rtas-size",

                                rtas_size);

    if (ret < 0) {

        fprintf(stderr, "Couldn't add rtas-size property: %s\n",

                fdt_strerror(ret));

        return ret;

    }



    for (i = 0; i < TOKEN_MAX; i++) {

        struct rtas_call *call = &rtas_table[i];



        if (!call->name) {

            continue;

        }



        ret = qemu_fdt_setprop_cell(fdt, "/rtas", call->name,

                                    i + TOKEN_BASE);

        if (ret < 0) {

            fprintf(stderr, "Couldn't add rtas token for %s: %s\n",

                    call->name, fdt_strerror(ret));

            return ret;

        }



    }

    return 0;

}
