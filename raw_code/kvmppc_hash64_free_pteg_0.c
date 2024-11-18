void kvmppc_hash64_free_pteg(uint64_t token)

{

    struct kvm_get_htab_buf *htab_buf;



    htab_buf = container_of((void *)(uintptr_t) token, struct kvm_get_htab_buf,

                            hpte);

    g_free(htab_buf);

    return;

}
