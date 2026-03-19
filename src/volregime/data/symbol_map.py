import yaml
from pathlib import Path

DEFAULT_OVERRIDES ={
    'BRK B': 'BRK-B',
    'BRK/B': 'BRK-B',
    'BF B': 'BF-B',
    'BF/B': 'BF-B'
}

class SymbolMap:
    def __init__(self, overrides_from_config = None, exclude=None):
        """
        Args:
            overrides_from_config: dict from symbols.yaml symbol_overrides section
            exclude: list of symbols to skip
        """
        self.overrides = {**DEFAULT_OVERRIDES}
        if overrides_from_config:
            self.overrides.update(overrides_from_config)
        self.exclude = set(exclude or [])

    def resolve(self, act_symbol):
        """
        map a dolthub act_symbol to a yfinance ticker
        returns (yf_ticker, should_skip)
        """
        act_symbol = act_symbol.strip()
        if act_symbol in self.exclude:
            return None, True
        
        if act_symbol in self.overrides:
            return self.overrides[act_symbol], False
        
        #default : assume act_symbol == yfinance ticker
        return act_symbol, False
    
    def resolve_all(self, act_symbols):
        """
        map a list of act_symbols
        returns a dict {act_symbol: yf_ticker} for valid symbols, and a list of skipped symbols
        """

        mapping = {}
        skipped = []
        for sym in act_symbols:
            yf_ticker, skip = self.resolve(sym)
            if skip:
                skipped.append(sym)
            else:
                mapping[sym] = yf_ticker
        return mapping, skipped
