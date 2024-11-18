int monitor_read_bdrv_key(BlockDriverState *bs)

{

    char password[256];

    int i;



    if (!bdrv_is_encrypted(bs))

        return 0;



    term_printf("%s (%s) is encrypted.\n", bdrv_get_device_name(bs),

                bdrv_get_encrypted_filename(bs));

    for(i = 0; i < 3; i++) {

        monitor_readline("Password: ", 1, password, sizeof(password));

        if (bdrv_set_key(bs, password) == 0)

            return 0;

        term_printf("invalid password\n");

    }

    return -EPERM;

}
