static int find_and_check_chardev(CharDriverState **chr,

                                  char *chr_name,

                                  Error **errp)

{

    CompareChardevProps props;



    *chr = qemu_chr_find(chr_name);

    if (*chr == NULL) {

        error_setg(errp, "Device '%s' not found",

                   chr_name);

        return 1;

    }



    memset(&props, 0, sizeof(props));

    if (qemu_opt_foreach((*chr)->opts, compare_chardev_opts, &props, errp)) {

        return 1;

    }



    if (!props.is_socket) {

        error_setg(errp, "chardev \"%s\" is not a tcp socket",

                   chr_name);

        return 1;

    }

    return 0;

}
