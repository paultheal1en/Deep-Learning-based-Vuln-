int qemu_read_config_file(const char *filename)

{

    FILE *f = fopen(filename, "r");

    int ret;



    if (f == NULL) {

        return -errno;

    }



    ret = qemu_config_parse(f, vm_config_groups, filename);

    fclose(f);



    if (ret == 0) {

        return 0;

    } else {

        return -EINVAL;

    }

}
