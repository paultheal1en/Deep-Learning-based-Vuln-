static void test_to_from_buf_1(void)

{

     unsigned niov;

     struct iovec *iov;

     size_t sz;

     unsigned char *ibuf, *obuf;

     unsigned i, j, n;



     iov_random(&iov, &niov);



     sz = iov_size(iov, niov);



     ibuf = g_malloc(sz + 8) + 4;

     memcpy(ibuf-4, "aaaa", 4); memcpy(ibuf + sz, "bbbb", 4);

     obuf = g_malloc(sz + 8) + 4;

     memcpy(obuf-4, "xxxx", 4); memcpy(obuf + sz, "yyyy", 4);



     /* fill in ibuf with 0123456... */

     for (i = 0; i < sz; ++i) {

         ibuf[i] = i & 255;

     }



     for (i = 0; i <= sz; ++i) {



         /* Test from/to buf for offset(i) in [0..sz] up to the end of buffer.

          * For last iteration with offset == sz, the procedure should

          * skip whole vector and process exactly 0 bytes */



         /* first set bytes [i..sz) to some "random" value */

         n = iov_memset(iov, niov, 0, 0xff, -1);

         g_assert(n == sz);



         /* next copy bytes [i..sz) from ibuf to iovec */

         n = iov_from_buf(iov, niov, i, ibuf + i, -1);

         g_assert(n == sz - i);



         /* clear part of obuf */

         memset(obuf + i, 0, sz - i);

         /* and set this part of obuf to values from iovec */

         n = iov_to_buf(iov, niov, i, obuf + i, -1);

         g_assert(n == sz - i);



         /* now compare resulting buffers */

         g_assert(memcmp(ibuf, obuf, sz) == 0);



         /* test just one char */

         n = iov_to_buf(iov, niov, i, obuf + i, 1);

         g_assert(n == (i < sz));

         if (n) {

             g_assert(obuf[i] == (i & 255));

         }



         for (j = i; j <= sz; ++j) {

             /* now test num of bytes cap up to byte no. j,

              * with j in [i..sz]. */



             /* clear iovec */

             n = iov_memset(iov, niov, 0, 0xff, -1);

             g_assert(n == sz);



             /* copy bytes [i..j) from ibuf to iovec */

             n = iov_from_buf(iov, niov, i, ibuf + i, j - i);

             g_assert(n == j - i);



             /* clear part of obuf */

             memset(obuf + i, 0, j - i);



             /* copy bytes [i..j) from iovec to obuf */

             n = iov_to_buf(iov, niov, i, obuf + i, j - i);

             g_assert(n == j - i);



             /* verify result */

             g_assert(memcmp(ibuf, obuf, sz) == 0);



             /* now actually check if the iovec contains the right data */

             test_iov_bytes(iov, niov, i, j - i);

         }

    }

    g_assert(!memcmp(ibuf-4, "aaaa", 4) && !memcmp(ibuf+sz, "bbbb", 4));

    g_free(ibuf-4);

    g_assert(!memcmp(obuf-4, "xxxx", 4) && !memcmp(obuf+sz, "yyyy", 4));

    g_free(obuf-4);

    iov_free(iov, niov);

}
