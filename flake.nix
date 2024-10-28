{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-unstable";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudnnSupoprt = true;
          };
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages (python-pkgs:
              with python-pkgs; [
                jupyter
                jupytext
                numpy
                scipy
                scikit-learn
                matplotlib
                tqdm
                pytest
                umap-learn # uniform manifold approximation
                # torch
                # torchvision
                # pytorch-lightning # import as `pytorch_lightning`
                # tensorboard
              ]))
            pkgs.cudaPackages.cudatoolkit
          ];
        };
      }
    );
}
