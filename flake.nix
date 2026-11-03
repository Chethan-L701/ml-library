{
  description = "machine learning library";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        name = "ml-library";
        ver = "0.1.0";
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = name;
          version = ver;
          src = self;

          buildPhase = "true";
          installPhase = ''
              mkdir -p $out/include/ml
              cp ${self}/include/* $out/include/ml
          '';

        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            # lang
            pkgs.gcc
            pkgs.pkg-config

            # build tools
            pkgs.cmake
            pkgs.ninja

            # dev tools
            pkgs.clang-tools
            pkgs.gdbgui

            # lsps
            pkgs.neocmakelsp
            pkgs.cmake-language-server
            pkgs.yaml-language-server
          ];
          NIX_DEV_SHELL = "${name}:${ver}";
        };
      }
    );
}
