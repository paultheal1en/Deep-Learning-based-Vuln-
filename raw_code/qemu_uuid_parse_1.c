int qemu_uuid_parse(const char *str, QemuUUID *uuid)

{

    unsigned char *uu = &uuid->data[0];

    int ret;



    if (strlen(str) != 36) {

        return -1;

    }



    ret = sscanf(str, UUID_FMT, &uu[0], &uu[1], &uu[2], &uu[3],

                 &uu[4], &uu[5], &uu[6], &uu[7], &uu[8], &uu[9],

                 &uu[10], &uu[11], &uu[12], &uu[13], &uu[14],

                 &uu[15]);



    if (ret != 16) {

        return -1;

    }

    return 0;

}
