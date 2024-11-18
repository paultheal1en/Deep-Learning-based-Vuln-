static inline void t_gen_zext(TCGv d, TCGv s, int size)

{

	if (size == 1)

		tcg_gen_ext8u_i32(d, s);

	else if (size == 2)

		tcg_gen_ext16u_i32(d, s);

	else if (GET_TCGV(d) != GET_TCGV(s))

		tcg_gen_mov_tl(d, s);

}
