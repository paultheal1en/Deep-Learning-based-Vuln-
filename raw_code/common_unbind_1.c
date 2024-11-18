static void common_unbind(struct common *c)

{

    xen_be_unbind_evtchn(&c->xendev);

    if (c->page) {

	munmap(c->page, XC_PAGE_SIZE);

	c->page = NULL;

    }

}
