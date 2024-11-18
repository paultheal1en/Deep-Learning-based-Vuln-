static inline int cris_bound_w(int v, int b)

{

	int r = v;

	asm ("bound.w\t%1, %0\n" : "+r" (r) : "ri" (b));

	return r;

}
