av_cold void ff_mlz_init_dict(void* context, MLZ *mlz) {

    mlz->dict = av_malloc_array(TABLE_SIZE, sizeof(*mlz->dict));



    mlz->flush_code            = FLUSH_CODE;

    mlz->current_dic_index_max = DIC_INDEX_INIT;

    mlz->dic_code_bit          = CODE_BIT_INIT;

    mlz->bump_code             = (DIC_INDEX_INIT - 1);

    mlz->next_code             = FIRST_CODE;

    mlz->freeze_flag           = 0;

    mlz->context               = context;

}
