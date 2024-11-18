static inline int compress_coef(int *coefs, int num)

{

    int i, res = 0;

    for (i = 0; i < num; i++)

        res += coef_test_compression(coefs[i]);

    return res == num ? 1 : 0;

}
