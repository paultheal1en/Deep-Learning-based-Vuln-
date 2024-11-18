static int vtd_dev_to_context_entry(IntelIOMMUState *s, uint8_t bus_num,

                                    uint8_t devfn, VTDContextEntry *ce)

{

    VTDRootEntry re;

    int ret_fr;



    ret_fr = vtd_get_root_entry(s, bus_num, &re);

    if (ret_fr) {

        return ret_fr;

    }



    if (!vtd_root_entry_present(&re)) {

        /* Not error - it's okay we don't have root entry. */

        trace_vtd_re_not_present(bus_num);

        return -VTD_FR_ROOT_ENTRY_P;

    } else if (re.rsvd || (re.val & VTD_ROOT_ENTRY_RSVD)) {

        trace_vtd_re_invalid(re.rsvd, re.val);

        return -VTD_FR_ROOT_ENTRY_RSVD;

    }



    ret_fr = vtd_get_context_entry_from_root(&re, devfn, ce);

    if (ret_fr) {

        return ret_fr;

    }



    if (!vtd_ce_present(ce)) {

        /* Not error - it's okay we don't have context entry. */

        trace_vtd_ce_not_present(bus_num, devfn);

        return -VTD_FR_CONTEXT_ENTRY_P;

    } else if ((ce->hi & VTD_CONTEXT_ENTRY_RSVD_HI) ||

               (ce->lo & VTD_CONTEXT_ENTRY_RSVD_LO)) {

        trace_vtd_ce_invalid(ce->hi, ce->lo);

        return -VTD_FR_CONTEXT_ENTRY_RSVD;

    }

    /* Check if the programming of context-entry is valid */

    if (!vtd_is_level_supported(s, vtd_ce_get_level(ce))) {

        trace_vtd_ce_invalid(ce->hi, ce->lo);

        return -VTD_FR_CONTEXT_ENTRY_INV;

    } else {

        switch (vtd_ce_get_type(ce)) {

        case VTD_CONTEXT_TT_MULTI_LEVEL:

            /* fall through */

        case VTD_CONTEXT_TT_DEV_IOTLB:

            break;

        default:

            trace_vtd_ce_invalid(ce->hi, ce->lo);

            return -VTD_FR_CONTEXT_ENTRY_INV;

        }

    }

    return 0;

}
