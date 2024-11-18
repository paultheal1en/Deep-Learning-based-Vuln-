static int kvm_irqchip_get_virq(KVMState *s)

{

    uint32_t *word = s->used_gsi_bitmap;

    int max_words = ALIGN(s->gsi_count, 32) / 32;

    int i, zeroes;

    bool retry = true;



again:

    /* Return the lowest unused GSI in the bitmap */

    for (i = 0; i < max_words; i++) {

        zeroes = ctz32(~word[i]);

        if (zeroes == 32) {

            continue;

        }



        return zeroes + i * 32;

    }

    if (!s->direct_msi && retry) {

        retry = false;

        kvm_flush_dynamic_msi_routes(s);

        goto again;

    }

    return -ENOSPC;



}
