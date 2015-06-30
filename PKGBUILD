# Maintainer: Eugene Dvoretsky <radioxoma at gmail.com>

pkgname=immunopy
pkgver=0.1
pkgrel=1
epoch=
pkgdesc="IHC image real time analyzer"
arch=('any')
url=""
license=('MIT')
depends=('python2-numpy' 'python2-scipy'
         'python2-imaging' 'python2-scikit-image' 'micromanager')
makedepends=('python2-setuptools')
checkdepends=()
optdepends=('python2-pyopencl: speed up with OpenCL support')
provides=()
conflicts=()
replaces=()
backup=()
options=()
install=
changelog=
source=("$pkgname::git+file:///home/radioxoma/dev/source/immunopy/"
        "immunopy.desktop")
noextract=()
sha256sums=('SKIP'
            'a45382a5e14fcfbceca2faacffc2aa52a2b32efe227a22b3c65abc3097ff4c49')

pkgver() {
  # For debugging
  # cp -r "/home/radioxoma/dev/source/immunopy" "$srcdir"
  cd "$pkgname"
  # git describe --long | sed -r 's/([^-]*-g)/r\1/;s/-/./g'
  # python2 setup.py --version
  git describe --tags | sed 's/^v//;s/\([^-]*-g\)/r\1/;s/-/./g'
}

# prepare() {
# 	# git clone file:///home/radioxoma/dev/source/assent
# 	# cd "$srcdir/$pkgname-$pkgver"
# }

# build() {
# 	cd "$srcdir/$pkgname-$pkgver"
# 	./configure --prefix=/usr
# 	make
# }

check() {
  cd "$srcdir/$pkgname"
  python2 setup.py test
}

package() {
  cd "$srcdir/$pkgname"
  python2 setup.py install --root="$pkgdir/" --optimize=1
  install -Dm644 "$srcdir"/immunopy.desktop "$pkgdir"/usr/share/applications/immunopy.desktop
  install -D "$srcdir/$pkgname/LICENSE" "$pkgdir"/usr/share/licenses/$pkgname/LICENSE
}
