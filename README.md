# Minimum Edge Set Perturbation Problem (MESP)

Currently, there are three main research heuristics:
- Homophily
- Degree
- Common Neighbors

See their findings here:
- [Homophily](https://github.com/wrcorcoran/minimum-edge-set-perturbation/blob/main/homophily/experiments/FINDINGS.md)
- [Degree](https://github.com/wrcorcoran/minimum-edge-set-perturbation/blob/main/degree/experiments/FINDINGS.md)
- [Common Neighbors](https://github.com/wrcorcoran/minimum-edge-set-perturbation/blob/main/common-neighbors/experiments/FINDINGS.md)

<!-- ```
NAVIGATE TO YOUR HOME DIRECTORY USING TERMINAL

wget -c https://ftp.gnu.org/gnu/glibc/glibc-2.27.tar.gz
tar -zxvf glibc-2.27.tar.gz
mkdir glibc-2.27/build

wget http://ftp.gnu.org/gnu/bison/bison-2.7.tar.gz
tar -xvzf bison-2.7.tar.gz

wget https://ftp.gnu.org/gnu/m4/m4-latest.tar.gz
tar -xvzf m4-latest.tar.gz
cd m4-1.4.19
./configure --prefix=/home/*USERNAME*/m4
make
make install

cd ~
cd bison-2.7
PATH=$PATH:/home/*USERNAME*/m4/bin/
./configure --prefix=/home/*USERNAME*/bison
make
make install
PATH=$PATH:/home/*USERNAME*/bison/bin/

cd ~
cd glibc-2.27/build
../configure --prefix=/opt/glibc
make && make install

NOTE: this step may hang for awhile, let it be!

PATH=$PATH:/home/*USERNAME*/glibc/bin/

or

conda install pytorch torchvision -c pytorch && conda install -c dglteam dgl

``` -->
