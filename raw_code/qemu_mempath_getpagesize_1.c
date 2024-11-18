size_t qemu_mempath_getpagesize(const char *mem_path)

{

#ifdef CONFIG_LINUX

    struct statfs fs;

    int ret;



    do {

        ret = statfs(mem_path, &fs);

    } while (ret != 0 && errno == EINTR);



    if (ret != 0) {

        fprintf(stderr, "Couldn't statfs() memory path: %s\n",

                strerror(errno));

        exit(1);

    }



    if (fs.f_type == HUGETLBFS_MAGIC) {

        /* It's hugepage, return the huge page size */

        return fs.f_bsize;

    }








    return getpagesize();

}