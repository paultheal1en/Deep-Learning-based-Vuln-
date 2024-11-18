static int kvm_irqchip_get_virq(KVMState *s)

{

    uint32_t *word = s->used_gsi_bitmap;

    int max_words = ALIGN(s->gsi_count, 32) / 32;

    int i, bit;

    bool retry = true;



again:

    /* Return the lowest unused GSI in the bitmap */

    for (i = 0; i < max_words; i++) {

        bit = ffs(~word[i]);

        if (!bit) {

            continue;

        }



        return bit - 1 + i * 32;

    }

    if (retry) {

        retry = false;

        kvm_flush_dynamic_msi_routes(s);

        goto again;

    }

    return -ENOSPC;



}
