static int file_open_dir(URLContext *h)

{

#if HAVE_DIRENT_H

    FileContext *c = h->priv_data;



    c->dir = opendir(h->filename);

    if (!c->dir)

        return AVERROR(errno);



    return 0;

#else

    return AVERROR(ENOSYS);

#endif /* HAVE_DIRENT_H */

}
