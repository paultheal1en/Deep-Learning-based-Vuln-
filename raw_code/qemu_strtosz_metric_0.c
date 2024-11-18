int64_t qemu_strtosz_metric(const char *nptr, char **end)

{

    return do_strtosz(nptr, end, 'B', 1000);

}
