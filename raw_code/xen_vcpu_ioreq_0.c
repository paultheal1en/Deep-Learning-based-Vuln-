static inline ioreq_t *xen_vcpu_ioreq(shared_iopage_t *shared_page, int vcpu)

{

    return &shared_page->vcpu_iodata[vcpu].vp_ioreq;

}
