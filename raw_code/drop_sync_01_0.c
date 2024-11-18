static int drop_sync(QIOChannel *ioc, size_t size, Error **errp)

{

    ssize_t ret = 0;

    char small[1024];

    char *buffer;



    buffer = sizeof(small) >= size ? small : g_malloc(MIN(65536, size));

    while (size > 0) {

        ssize_t count = MIN(65536, size);

        ret = read_sync(ioc, buffer, MIN(65536, size), errp);



        if (ret < 0) {

            goto cleanup;

        }

        size -= count;

    }



 cleanup:

    if (buffer != small) {

        g_free(buffer);

    }

    return ret;

}
