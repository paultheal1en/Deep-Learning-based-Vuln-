static void rtas_ibm_configure_connector(PowerPCCPU *cpu,

                                         sPAPRMachineState *spapr,

                                         uint32_t token, uint32_t nargs,

                                         target_ulong args, uint32_t nret,

                                         target_ulong rets)

{

    uint64_t wa_addr;

    uint64_t wa_offset;

    uint32_t drc_index;

    sPAPRDRConnector *drc;

    sPAPRDRConnectorClass *drck;

    sPAPRConfigureConnectorState *ccs;

    sPAPRDRCCResponse resp = SPAPR_DR_CC_RESPONSE_CONTINUE;

    int rc;



    if (nargs != 2 || nret != 1) {

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;

    }



    wa_addr = ((uint64_t)rtas_ld(args, 1) << 32) | rtas_ld(args, 0);



    drc_index = rtas_ld(wa_addr, 0);

    drc = spapr_drc_by_index(drc_index);

    if (!drc) {

        trace_spapr_rtas_ibm_configure_connector_invalid(drc_index);

        rc = RTAS_OUT_PARAM_ERROR;

        goto out;

    }



    if ((drc->state != SPAPR_DRC_STATE_LOGICAL_UNISOLATE)

        && (drc->state != SPAPR_DRC_STATE_PHYSICAL_UNISOLATE)) {

        /* Need to unisolate the device before configuring */

        rc = SPAPR_DR_CC_RESPONSE_NOT_CONFIGURABLE;

        goto out;

    }



    g_assert(drc->fdt);



    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);



    ccs = drc->ccs;

    if (!ccs) {

        ccs = g_new0(sPAPRConfigureConnectorState, 1);

        ccs->fdt_offset = drc->fdt_start_offset;

        drc->ccs = ccs;

    }



    do {

        uint32_t tag;

        const char *name;

        const struct fdt_property *prop;

        int fdt_offset_next, prop_len;



        tag = fdt_next_tag(drc->fdt, ccs->fdt_offset, &fdt_offset_next);



        switch (tag) {

        case FDT_BEGIN_NODE:

            ccs->fdt_depth++;

            name = fdt_get_name(drc->fdt, ccs->fdt_offset, NULL);



            /* provide the name of the next OF node */

            wa_offset = CC_VAL_DATA_OFFSET;

            rtas_st(wa_addr, CC_IDX_NODE_NAME_OFFSET, wa_offset);

            configure_connector_st(wa_addr, wa_offset, name, strlen(name) + 1);

            resp = SPAPR_DR_CC_RESPONSE_NEXT_CHILD;

            break;

        case FDT_END_NODE:

            ccs->fdt_depth--;

            if (ccs->fdt_depth == 0) {

                uint32_t drc_index = spapr_drc_index(drc);



                /* done sending the device tree, move to configured state */

                trace_spapr_drc_set_configured(drc_index);

                drc->state = drck->ready_state;

                g_free(ccs);

                drc->ccs = NULL;

                ccs = NULL;

                resp = SPAPR_DR_CC_RESPONSE_SUCCESS;

            } else {

                resp = SPAPR_DR_CC_RESPONSE_PREV_PARENT;

            }

            break;

        case FDT_PROP:

            prop = fdt_get_property_by_offset(drc->fdt, ccs->fdt_offset,

                                              &prop_len);

            name = fdt_string(drc->fdt, fdt32_to_cpu(prop->nameoff));



            /* provide the name of the next OF property */

            wa_offset = CC_VAL_DATA_OFFSET;

            rtas_st(wa_addr, CC_IDX_PROP_NAME_OFFSET, wa_offset);

            configure_connector_st(wa_addr, wa_offset, name, strlen(name) + 1);



            /* provide the length and value of the OF property. data gets

             * placed immediately after NULL terminator of the OF property's

             * name string

             */

            wa_offset += strlen(name) + 1,

            rtas_st(wa_addr, CC_IDX_PROP_LEN, prop_len);

            rtas_st(wa_addr, CC_IDX_PROP_DATA_OFFSET, wa_offset);

            configure_connector_st(wa_addr, wa_offset, prop->data, prop_len);

            resp = SPAPR_DR_CC_RESPONSE_NEXT_PROPERTY;

            break;

        case FDT_END:

            resp = SPAPR_DR_CC_RESPONSE_ERROR;

        default:

            /* keep seeking for an actionable tag */

            break;

        }

        if (ccs) {

            ccs->fdt_offset = fdt_offset_next;

        }

    } while (resp == SPAPR_DR_CC_RESPONSE_CONTINUE);



    rc = resp;

out:

    rtas_st(rets, 0, rc);

}
