static inline int decode_cabac_mb_transform_size( H264Context *h ) {

    return get_cabac( &h->cabac, &h->cabac_state[399 + h->neighbor_transform_size] );

}
