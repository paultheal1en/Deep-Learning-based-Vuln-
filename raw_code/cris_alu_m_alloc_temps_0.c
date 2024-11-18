static inline void cris_alu_m_alloc_temps(TCGv *t)

{

	t[0] = tcg_temp_new(TCG_TYPE_TL);

	t[1] = tcg_temp_new(TCG_TYPE_TL);

}
