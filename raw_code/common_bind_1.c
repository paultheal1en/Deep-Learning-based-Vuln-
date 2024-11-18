static int common_bind(struct common *c)

{

    uint64_t mfn;



    if (xenstore_read_fe_uint64(&c->xendev, "page-ref", &mfn) == -1)

	return -1;

    assert(mfn == (xen_pfn_t)mfn);



    if (xenstore_read_fe_int(&c->xendev, "event-channel", &c->xendev.remote_port) == -1)

	return -1;



    c->page = xc_map_foreign_range(xen_xc, c->xendev.dom,

				   XC_PAGE_SIZE,

				   PROT_READ | PROT_WRITE, mfn);

    if (c->page == NULL)

	return -1;



    xen_be_bind_evtchn(&c->xendev);

    xen_be_printf(&c->xendev, 1, "ring mfn %"PRIx64", remote-port %d, local-port %d\n",

		  mfn, c->xendev.remote_port, c->xendev.local_port);



    return 0;

}
