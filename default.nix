let
  pkgs = import <nixpkgs> {};
in with pkgs; stdenv.mkDerivation rec {
  name = "rust-gpu";

  # Workaround for https://github.com/NixOS/nixpkgs/issues/60919.
  hardeningDisable = [ "fortify" ];

  # Allow cargo to download crates.
  SSL_CERT_FILE = "${cacert}/etc/ssl/certs/ca-bundle.crt";

  buildInputs = [
    pkgconfig rustup x11 libxkbcommon
  ];

  # Runtime dependencies.
  LD_LIBRARY_PATH = with xlibs; stdenv.lib.makeLibraryPath [
    libX11 libXcursor libXi libXrandr vulkan-loader
  ];

  # swiftshader_env =
  #   let swiftshader = pkgs.swiftshader.overrideAttrs (old: {
  #     version = "2020-11-20";
  #     src = fetchgit {
  #       url = "https://swiftshader.googlesource.com/SwiftShader";
  #       rev = "6d612051c083238db89541be4fbb2d624a9baef4";
  #       sha256 = "0xq317mgmbs2pzq84vvfn0nqjd38djh5jzrkyixfxkxpaaz5j6s8";
  #     };
  #     patches = [
  #       (writeText "no-validation.patch" ''
  #         --- a/src/Vulkan/VkPipeline.cpp
  #         +++ b/src/Vulkan/VkPipeline.cpp
  #         @@ -78,1 +78,1 @@
  #         -	opt.Run(code.data(), code.size(), &optimized);
  #         +	opt.Run(code.data(), code.size(), &optimized, spvtools::ValidatorOptions(), true);
  #         '')
  #     ];
  #     buildInputs = old.buildInputs ++ [zlib];
  #     hardeningDisable = [ "all" ];
  #     cmakeFlags = [
  #       "-DCMAKE_BUILD_TYPE=Debug"
  #       "-DSWIFTSHADER_WARNINGS_AS_ERRORS=0"
  #       "-DCMAKE_CXX_FLAGS=-DNDEBUG -O0"
  #       # "-DSWIFTSHADER_LOGGING_LEVEL=Verbose"
  #     ];
  #     dontStrip = true;
  #   });
  #   in "VK_ICD_FILENAMES=${swiftshader}/share/vulkan/icd.d/vk_swiftshader_icd.json";
}
