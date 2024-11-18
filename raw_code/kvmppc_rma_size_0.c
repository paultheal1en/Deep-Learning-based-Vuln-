uint64_t kvmppc_rma_size(uint64_t current_size, unsigned int hash_shift)

{

    if (cap_ppc_rma >= 2) {

        return current_size;

    }

    return MIN(current_size,

               getrampagesize() << (hash_shift - 7));

}
