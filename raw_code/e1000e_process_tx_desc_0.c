e1000e_process_tx_desc(E1000ECore *core,

                       struct e1000e_tx *tx,

                       struct e1000_tx_desc *dp,

                       int queue_index)

{

    uint32_t txd_lower = le32_to_cpu(dp->lower.data);

    uint32_t dtype = txd_lower & (E1000_TXD_CMD_DEXT | E1000_TXD_DTYP_D);

    unsigned int split_size = txd_lower & 0xffff;

    uint64_t addr;

    struct e1000_context_desc *xp = (struct e1000_context_desc *)dp;

    bool eop = txd_lower & E1000_TXD_CMD_EOP;



    if (dtype == E1000_TXD_CMD_DEXT) { /* context descriptor */

        e1000x_read_tx_ctx_descr(xp, &tx->props);

        e1000e_process_snap_option(core, le32_to_cpu(xp->cmd_and_length));

        return;

    } else if (dtype == (E1000_TXD_CMD_DEXT | E1000_TXD_DTYP_D)) {

        /* data descriptor */

        tx->props.sum_needed = le32_to_cpu(dp->upper.data) >> 8;

        tx->props.cptse = (txd_lower & E1000_TXD_CMD_TSE) ? 1 : 0;

        e1000e_process_ts_option(core, dp);

    } else {

        /* legacy descriptor */

        e1000e_process_ts_option(core, dp);

        tx->props.cptse = 0;

    }



    addr = le64_to_cpu(dp->buffer_addr);



    if (!tx->skip_cp) {

        if (!net_tx_pkt_add_raw_fragment(tx->tx_pkt, addr, split_size)) {

            tx->skip_cp = true;

        }

    }



    if (eop) {

        if (!tx->skip_cp && net_tx_pkt_parse(tx->tx_pkt)) {

            if (e1000x_vlan_enabled(core->mac) &&

                e1000x_is_vlan_txd(txd_lower)) {

                net_tx_pkt_setup_vlan_header_ex(tx->tx_pkt,

                    le16_to_cpu(dp->upper.fields.special), core->vet);

            }

            if (e1000e_tx_pkt_send(core, tx, queue_index)) {

                e1000e_on_tx_done_update_stats(core, tx->tx_pkt);

            }

        }



        tx->skip_cp = false;

        net_tx_pkt_reset(tx->tx_pkt);



        tx->props.sum_needed = 0;

        tx->props.cptse = 0;

    }

}
