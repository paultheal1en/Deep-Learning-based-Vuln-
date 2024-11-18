static void GCC_FMT_ATTR(2, 3) blkverify_err(BlkverifyAIOCB *acb,

                                             const char *fmt, ...)

{

    va_list ap;



    va_start(ap, fmt);

    fprintf(stderr, "blkverify: %s sector_num=%" PRId64 " nb_sectors=%d ",

            acb->is_write ? "write" : "read", acb->sector_num,

            acb->nb_sectors);

    vfprintf(stderr, fmt, ap);

    fprintf(stderr, "\n");

    va_end(ap);

    exit(1);

}
