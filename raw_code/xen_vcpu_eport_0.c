static inline uint32_t xen_vcpu_eport(shared_iopage_t *shared_page, int i)

{

    return shared_page->vcpu_iodata[i].vp_eport;

}
