void g_free(void *ptr)

{

    /* FIXME: We should unmark the reserved pages here.  However this gets

       complicated when one target page spans multiple host pages, so we

       don't bother.  */

    size_t *p;

    p = (size_t *)((char *)ptr - 16);

    munmap(p, *p);

}
