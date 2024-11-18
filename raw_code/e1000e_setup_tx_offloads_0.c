e1000e_setup_tx_offloads(E1000ECore *core, struct e1000e_tx *tx)

{

    if (tx->props.tse && tx->props.cptse) {

        net_tx_pkt_build_vheader(tx->tx_pkt, true, true, tx->props.mss);

        net_tx_pkt_update_ip_checksums(tx->tx_pkt);

        e1000x_inc_reg_if_not_full(core->mac, TSCTC);

        return;

    }



    if (tx->props.sum_needed & E1000_TXD_POPTS_TXSM) {

        net_tx_pkt_build_vheader(tx->tx_pkt, false, true, 0);

    }



    if (tx->props.sum_needed & E1000_TXD_POPTS_IXSM) {

        net_tx_pkt_update_ip_hdr_checksum(tx->tx_pkt);

    }

}
