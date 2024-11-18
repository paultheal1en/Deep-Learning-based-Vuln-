static bool vtd_process_inv_desc(IntelIOMMUState *s)

{

    VTDInvDesc inv_desc;

    uint8_t desc_type;



    VTD_DPRINTF(INV, "iq head %"PRIu16, s->iq_head);

    if (!vtd_get_inv_desc(s->iq, s->iq_head, &inv_desc)) {

        s->iq_last_desc_type = VTD_INV_DESC_NONE;

        return false;

    }

    desc_type = inv_desc.lo & VTD_INV_DESC_TYPE;

    /* FIXME: should update at first or at last? */

    s->iq_last_desc_type = desc_type;



    switch (desc_type) {

    case VTD_INV_DESC_CC:

        VTD_DPRINTF(INV, "Context-cache Invalidate Descriptor hi 0x%"PRIx64

                    " lo 0x%"PRIx64, inv_desc.hi, inv_desc.lo);

        if (!vtd_process_context_cache_desc(s, &inv_desc)) {

            return false;

        }

        break;



    case VTD_INV_DESC_IOTLB:

        VTD_DPRINTF(INV, "IOTLB Invalidate Descriptor hi 0x%"PRIx64

                    " lo 0x%"PRIx64, inv_desc.hi, inv_desc.lo);

        if (!vtd_process_iotlb_desc(s, &inv_desc)) {

            return false;

        }

        break;



    case VTD_INV_DESC_WAIT:

        VTD_DPRINTF(INV, "Invalidation Wait Descriptor hi 0x%"PRIx64

                    " lo 0x%"PRIx64, inv_desc.hi, inv_desc.lo);

        if (!vtd_process_wait_desc(s, &inv_desc)) {

            return false;

        }

        break;



    case VTD_INV_DESC_IEC:

        VTD_DPRINTF(INV, "Interrupt Entry Cache Invalidation "

                    "not implemented yet");

        /*

         * Since currently we do not cache interrupt entries, we can

         * just mark this descriptor as "good" and move on.

         */

        break;



    default:

        VTD_DPRINTF(GENERAL, "error: unkonw Invalidation Descriptor type "

                    "hi 0x%"PRIx64 " lo 0x%"PRIx64 " type %"PRIu8,

                    inv_desc.hi, inv_desc.lo, desc_type);

        return false;

    }

    s->iq_head++;

    if (s->iq_head == s->iq_size) {

        s->iq_head = 0;

    }

    return true;

}
