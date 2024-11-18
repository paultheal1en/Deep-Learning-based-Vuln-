static ssize_t drop_sync(QIOChannel *ioc, size_t size)

{

    ssize_t ret, dropped = size;

    char small[1024];

    char *buffer;



    buffer = sizeof(small) < size ? small : g_malloc(MIN(65536, size));

    while (size > 0) {

        ret = read_sync(ioc, buffer, MIN(65536, size));

        if (ret < 0) {

            goto cleanup;

        }

        assert(ret <= size);

        size -= ret;

    }

    ret = dropped;



 cleanup:

    if (buffer != small) {

        g_free(buffer);

    }

    return ret;

}
